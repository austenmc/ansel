#!/bin/bash

source lib/lock.sh

bar() {
  echo $1 $2 $3
}

test() {
  one="$1"
  two="$2"
    echo $one $two
    sleep 5
}

callback() {
  echo callback: $2
}

rm -Rf /tmp/foo
lock /tmp/foo callback test one two &
lock /tmp/foo callback test three four &
