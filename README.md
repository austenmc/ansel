# ansel

Lightweight scripts to manage an automatic photo download and archive workflow from camera to local storage, using Toshiba FlashAir SD cards.

Features:

* Lists remote files on a Toshia FlashAir SD card using their [API](https://flashair-developers.com/en/documents/api/configcgi/).
* Syncs files between a remote FlashAir and a local directory. Only syncs top-level directory; does not recurse into remote directories. Syncing currently means downloading files to a local folder `RAW-yyyy-mm` using the modification date of the remote file. This is so I can more easily manage these large folders using Dropbox.
* Can send messages using an FB Messenger bot.

Yes, this is hodgepodge of stuff, but it's what I need.

# Installation

```
yarn install
```

# Usage

```
$ ansel --help

  Usage: ansel [options] [command]

  Lightweight scripts to manage an automatic photo workflow from Toshiba FlashAir to local storage.


  Options:

    -V, --version  output the version number
    -h, --help     output usage information


  Commands:

    remote-listing <host> <directory>      List contents of remote directory
    local-listing <directory>              List contents of local directory
    sync-files                             Sync contents of remote directory to local directory
    paths                                  Extract file paths from an Ansel JSON listing
    send-message <psid> <token> <message>  Send simple message via FB Messenger bot
    send-image <psid> <token> <file>       Send image attachment via FB Messenger bot
    help [cmd]                             display help for [cmd]
```

# Development

This app is based on a generic NodeJS CLI tmeplate. Some batteries included:

+ ES6 + babel,
  + Removes Flow type annotations,
  + Transforms imports to lazy CommonJS requires,
  + Transforms async/await to generators,
+ [ESLint][eslint] with the [airbnb-base][airbnb-base] and [flowtype][eslint-flowtype] rules,
+ [Jest][jest] unit testing and coverage,
+ [Type definitions][flow-typed] for Jest,
+ [NPM scripts for common operations](#available-scripts),
+ [.editorconfig][editorconfig] for consistent file format,

## Available scripts

Run using `yarn run <script>` comand.

+ `clean` - remove coverage data, Jest cache and transpiled files,
+ `lint` - lint source files and tests,
+ `typecheck` - check type annotations,
+ `test` - lint, typecheck and run tests with coverage,
+ `test-only` - run tests with coverage,
+ `test:watch` - interactive watch mode to automatically re-run tests,
+ `build` - compile source files,
+ `build:watch` - interactive watch mode, compile sources on change.
