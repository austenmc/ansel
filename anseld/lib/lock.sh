#!/bin/bash
#
# lock path/to/lockdir callback function arguments
#
# callback: $1=retvalue $2=output
# Note that lockdir/.. needs to exist.
#

lock() {
  PID=$(sh -c 'echo $PPID')
  LOCKDIR="$1"
  CALLBACK="$2"
  FUNC="$3"
  ARGUMENTS=( "${@:4}" )

  # lock dirs/files
  PIDFILE="${LOCKDIR}/pid"

  # relevant exit codes
  ENO_LOCKFAIL=2;
  ENO_RECVSIG=3;

  ###
  ### start locking attempt
  ###

  if mkdir "${LOCKDIR}" &>/dev/null; then
    # lock succeeded, install signal handlers before storing the PID just in case
    # storing the PID fails
    trap 'rm -rf "${LOCKDIR}"' 0
    echo "${PID}" >"${PIDFILE}"
    # the following handler will exit the script upon receiving these signals
    # the trap on "0" (EXIT) from above will be triggered by this trap's "exit" command!
    trap 'exit ${ENO_RECVSIG}' 1 2 3 15
  else
    # lock failed, check if the other PID is alive

    # if cat isn't able to read the file, another instance is probably
    # about to remove the lock -- exit, we're *still* locked
    #  Thanks to Grzegorz Wierzowiecki for pointing out this race condition on
    #  http://wiki.grzegorz.wierzowiecki.pl/code:mutex-in-bash
    if ! OTHERPID=$(cat "${PIDFILE}")
    then
      exit ${ENO_LOCKFAIL}
    fi

    if ! kill -0 "$OTHERPID" &>/dev/null
    then
      # lock is stale, remove it and restart
      rm -rf "${LOCKDIR}"
      exec "$0" "$@"
    else
      # lock is valid and OTHERPID is active - exit, we're locked!
      exit ${ENO_LOCKFAIL}
    fi
  fi

  OUTPUT=$(${FUNC} "${ARGUMENTS[@]}")
  ${CALLBACK} $? "$OUTPUT"
}
