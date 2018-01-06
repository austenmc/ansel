#!/usr/bin/env node
/** @flow */
const path = require('path');

const pkg = require(path.join(__dirname, '..', 'package.json'));
const program = require('commander');

program
  .version(pkg.version)
  .description(pkg.description)
  .command('remote-listing <host> <directory>', 'List contents of remote directory')
  .command('local-listing <directory>', 'List contents of local directory')
  .command('sync-files', 'Sync contents of remote directory to local directory')
  .command('paths', 'Extract file paths from an Ansel JSON listing')
  .command('send-message <psid> <token> <message>', 'Send simple message via FB Messenger bot')
  .command('send-image <psid> <token> <file>', 'Send image attachment via FB Messenger bot')
  .parse(process.argv);
