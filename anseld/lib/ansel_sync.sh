#!/bin/bash
#
# ansel_sync host remote/directory local/directory
#

ansel_sync() {
  host="$1"
  remote_directory="$2"
  local_directory="$3"

  # Only continue if host is there.
  if ping -c 1 "${host}" &> /dev/null
  then
    # Gather remote and local listing.
    log "Starting sync from ${host}..."

    if ! remote_listing=$(ansel remote-listing "${host}" "${remote_directory}")
    then
      echo -e "Remote listing failed:\\n${remote_listing}";
      return 1;
    fi

    if ! local_listing=$(ansel local-list "${local_directory}")
    then
      echo -e "Local listing failed:\\n${local_listing}";
      return 1;
    fi

    sync_input="{ \"remote\": ${remote_listing}, \"local\": ${local_listing} }"
    log "Sync input:"
    log "${sync_input}"

    # Sync remote and local directories.
    if ! sync_output=$(echo "${sync_input}" | ansel sync-files)
    then
      echo -e "File sync failed:\\n${sync_output}";
      return 1;
    fi

    if [[ -z "${sync_output}" ]]
    then
      log "No output from sync"
      return 0;
    fi
    log "Sync ouput:"
    log "${sync_output}"

    successfully_synced=$(echo "${sync_output}" | ansel paths --status=ok)
    failed_synced=$(echo "${sync_output}" | ansel paths --status=failed)

    successfully_synced_count=$(echo "${successfully_synced}" | wc -l)
    failed_synced_count=$(echo "${failed_synced}" | wc -l)

    if [[ "${successfully_synced_count}" != 0 ]]
    then
      echo -e "Succesfully synced ${successfully_synced_count} files.";
    fi

    if [[ "${failed_synced_count}" != 0 ]]
    then
      echo -e "Failed to sync ${failed_synced_count} files:\\n${failed_synced}";
      return 1
    fi

    # Convert RAW files to JPG
    synced_raw_count=0
    successfully_converted_count=0
    while read -r line
    do
      (( synced_raw_count++ ))
      dir=$(dirname "${line}")
      name=$(basename "${line}")
      output="${dir}/{$name%.}.jpg"
      if sips -s format jpeg "${line}" --out "${output}"
      then
        (( successfully_converted_count++ ))
        log "...${output}"
      fi
    done <<< "echo ${successfully_synced} | grep .RAW"

    # Spot check number of created jpg files...
    if (( successfully_converted_count != synced_raw_count ))
    then
      echo -e "Only converted ${successfully_converted_count} / ${synced_raw_count} RAW files."
    else
      echo -e "Successfully converted ${successfully_converted_count} RAW files."
    fi
  else
    log "Couldn't reach ${host} for sync"
  fi

  return 0
}
