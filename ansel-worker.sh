#!/bin/bash
#
ansel=$1
host=$2
remote_directory=$3
local_directory=$4

# Sender ID for FB Messenger.
psid=$5

# Auth token for FB Messenger.
token=$6

# Mutex feature from http://wiki.bash-hackers.org/howto/mutex

# lock dirs/files
LOCKDIR="/var/lock/ansel"
PIDFILE="${LOCKDIR}/PID"

# exit codes and text
ENO_SUCCESS=0; ETXT[0]="ENO_SUCCESS"
ENO_GENERAL=1; ETXT[1]="ENO_GENERAL"
ENO_LOCKFAIL=2; ETXT[2]="ENO_LOCKFAIL"
ENO_RECVSIG=3; ETXT[3]="ENO_RECVSIG"

###
### start locking attempt
###

if mkdir "${LOCKDIR}" &>/dev/null; then
  # lock succeeded, install signal handlers before storing the PID just in case
  # storing the PID fails
  trap 'rm -rf "${LOCKDIR}"' 0
  echo "$$" >"${PIDFILE}"
  # the following handler will exit the script upon receiving these signals
  # the trap on "0" (EXIT) from above will be triggered by this trap's "exit" command!
  trap 'echo "[statsgen] Killed by a signal." >&2
        exit ${ENO_RECVSIG}' 1 2 3 15
else
  # lock failed, check if the other PID is alive
  OTHERPID="$(cat "${PIDFILE}")"

  # if cat isn't able to read the file, another instance is probably
  # about to remove the lock -- exit, we're *still* locked
  #  Thanks to Grzegorz Wierzowiecki for pointing out this race condition on
  #  http://wiki.grzegorz.wierzowiecki.pl/code:mutex-in-bash
  if [ $? != 0 ]
  then
    exit ${ENO_LOCKFAIL}
  fi

  if ! kill -0 $OTHERPID &>/dev/null
  then
    # lock is stale, remove it and restart
    echo "removing stale lock of nonexistant PID ${OTHERPID}" >&2
    rm -rf "${LOCKDIR}"
    echo "restarting..." >&2
    exec "$0" "$@"
  else
    # lock is valid and OTHERPID is active - exit, we're locked!
    exit ${ENO_LOCKFAIL}
  fi
fi

function send_preview
    exiv2 -ep1 -l "$2" "$1"
    filename=$(basename $1)
    preview="$2/${filename%.*}-preview1.jpg"
    if [[ -e "$preview" ]]
    then
      $ansel send-image $psid $token $preview
    else
      echo "Error: could not generate preview for $1"
    fi
end

# Only continue if host is there.
if ping -c 1 "$host" &> /dev/null
then
  # Gather remote and local listing.
  remote_listing=$($ansel remote-listing $host $remote_directory)
  if [[ $? != 0 ]]; then echo -e "Remote listing failed:\\n$remote_listing"; exit 1; fi

  local_listing=$($ansel local-list $local_directory)
  if [[ $? != 0 ]]; then echo -e "Local listing failed:\\n$local_listing"; exit 1; fi

  sync_input="{ \"remote\": $remote_listing, \"local\": $local_listing }"

  # Sync remote and local directories.
  sync_output=$(echo $sync_input | $ansel sync-files)
  if [[ $? != 0 ]]; then echo -e "File sync failed:\\n$sync_output"; exit 1; fi

  successfully_synced=$(echo $sync_output | $ansel paths --status=OK)
  failed_synced=$(echo $sync_output | $ansel paths --status=FAILED)

  successfully_synced_count=$(echo $successfully_synced | wc -l)
  failed_synced_count=$(echo $failed_synced | wc -l)
  if [[ $successfully_synced_count != 0 ]]
  then
    # Extract preview of first and last images and send them via Messenger.
    temp_dir=`mktemp -d -t ANSEL`
    first=$(echo $successfully_synced | head -1)
    last=$(echo $successfully_synced | tail -1)

    send_preivew "$first" "$temp_dir"
    send_preivew "$last" "$temp_dir"
  fi

  if [[ $failed_synced_count != 0 ]]
  then
    echo -e "Error: failed to sync $failed_synced_count files:\\n$failed_synced";
  fi
fi
