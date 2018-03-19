/** @flow */

/* eslint-disable no-use-before-define */

export type FileType = 'directory' | 'file';

export type FileStatus = 'ok' | 'failed';

export type Directory = {
  type: 'directory',
  name: string,
  path: string,
  contents: Listing,
};

export type File = {
  type: 'file',
  name: string,
  path: string,
  size: number,
  modified: Date,
  status?: string,
};

export type Listing = {
  [string]: File | Directory
};

export type FileListing = {
  [string]: File
};

export type DirectoryListing = {
  [string]: Directory
};

export type SyncListing = {
  remote: DirectoryListing,
  local: DirectoryListing,
};

export type Message = {
  psid: string,
  timestamp: number,
  message: string,
}

export type Messages = {
  messages: Array<Message>,
}

/* eslint-enable no-use-before-define */
