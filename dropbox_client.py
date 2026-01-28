"""Dropbox API wrapper with OAuth2 authentication."""

import dropbox
from dropbox import DropboxOAuth2Flow
from dropbox.exceptions import AuthError

import config


class DropboxClient:
    """Wrapper for Dropbox API with token management."""

    def __init__(self):
        self._dbx = None
        self._oauth_flow = None

    def is_authenticated(self):
        """Check if we have valid tokens."""
        tokens = config.load_tokens()
        if not tokens:
            return False

        # Try to use the tokens
        try:
            self._init_client(tokens["access_token"], tokens.get("refresh_token"))
            self._dbx.users_get_current_account()
            return True
        except AuthError:
            return False
        except Exception:
            return False

    def _init_client(self, access_token, refresh_token=None):
        """Initialize Dropbox client with tokens."""
        app_key, app_secret = config.get_dropbox_credentials()

        if refresh_token and app_key and app_secret:
            self._dbx = dropbox.Dropbox(
                oauth2_access_token=access_token,
                oauth2_refresh_token=refresh_token,
                app_key=app_key,
                app_secret=app_secret,
            )
        else:
            self._dbx = dropbox.Dropbox(oauth2_access_token=access_token)

    def get_auth_url(self):
        """Get OAuth authorization URL for user to visit."""
        app_key, app_secret = config.get_dropbox_credentials()
        if not app_key or not app_secret:
            raise ValueError(
                "Dropbox credentials not configured. "
                "Set DROPBOX_APP_KEY and DROPBOX_APP_SECRET environment variables "
                "or configure via the settings page."
            )

        self._oauth_flow = DropboxOAuth2Flow(
            consumer_key=app_key,
            consumer_secret=app_secret,
            redirect_uri=config.OAUTH_REDIRECT_URI,
            session={},
            csrf_token_session_key="dropbox-auth-csrf-token",
            token_access_type="offline",
        )
        return self._oauth_flow.start()

    def complete_auth(self, query_params, session_csrf):
        """Complete OAuth flow with callback parameters."""
        app_key, app_secret = config.get_dropbox_credentials()

        # Recreate the flow with the stored CSRF token
        self._oauth_flow = DropboxOAuth2Flow(
            consumer_key=app_key,
            consumer_secret=app_secret,
            redirect_uri=config.OAUTH_REDIRECT_URI,
            session={"dropbox-auth-csrf-token": session_csrf},
            csrf_token_session_key="dropbox-auth-csrf-token",
            token_access_type="offline",
        )

        result = self._oauth_flow.finish(query_params)

        # Save tokens
        tokens = {
            "access_token": result.access_token,
            "refresh_token": result.refresh_token,
            "account_id": result.account_id,
            "user_id": result.user_id,
        }
        config.save_tokens(tokens)

        # Initialize client
        self._init_client(result.access_token, result.refresh_token)
        return True

    def get_client(self):
        """Get the Dropbox client, initializing if needed."""
        if self._dbx is None:
            tokens = config.load_tokens()
            if tokens:
                self._init_client(tokens["access_token"], tokens.get("refresh_token"))
        return self._dbx

    def list_folder_recursive(self, path="", callback=None):
        """
        Recursively list all files in a folder.

        Args:
            path: Dropbox path to start from (empty string for root)
            callback: Optional callback function called with each batch of entries

        Yields:
            FileMetadata entries for each file found
        """
        dbx = self.get_client()
        if not dbx:
            raise RuntimeError("Not authenticated with Dropbox")

        try:
            result = dbx.files_list_folder(path, recursive=True)

            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        yield entry

                if callback:
                    callback(len(result.entries))

                if not result.has_more:
                    break

                result = dbx.files_list_folder_continue(result.cursor)

        except dropbox.exceptions.ApiError as e:
            raise RuntimeError(f"Dropbox API error: {e}")

    def download_file(self, path):
        """
        Download a file from Dropbox.

        Args:
            path: Dropbox path to the file

        Returns:
            Tuple of (metadata, file_content_bytes)
        """
        dbx = self.get_client()
        if not dbx:
            raise RuntimeError("Not authenticated with Dropbox")

        metadata, response = dbx.files_download(path)
        return metadata, response.content

    def get_temporary_link(self, path):
        """
        Get a temporary direct link to a Dropbox file (valid ~4 hours).

        Args:
            path: Dropbox path to the file

        Returns:
            Temporary URL string
        """
        dbx = self.get_client()
        if not dbx:
            raise RuntimeError("Not authenticated with Dropbox")

        result = dbx.files_get_temporary_link(path)
        return result.link

    def list_folders(self, path=""):
        """List folders at a given path."""
        dbx = self.get_client()
        if not dbx:
            raise RuntimeError("Not authenticated with Dropbox")

        folders = []
        try:
            result = dbx.files_list_folder(path)
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FolderMetadata):
                    folders.append({
                        "path": entry.path_display,
                        "name": entry.name,
                    })
            folders.sort(key=lambda f: f["name"].lower())
        except dropbox.exceptions.ApiError as e:
            raise RuntimeError(f"Dropbox API error: {e}")

        return folders

    def get_account_info(self):
        """Get current account information."""
        dbx = self.get_client()
        if not dbx:
            return None

        try:
            account = dbx.users_get_current_account()
            return {
                "name": account.name.display_name,
                "email": account.email,
            }
        except Exception:
            return None


# Global instance
dropbox_client = DropboxClient()
