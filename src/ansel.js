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
  .command('get-messages <psid>', 'Get pending messages via FB Messenger bot')
  .command('messages', 'Extract messages text from a messages JSON listing')
  .command('delete-messages <psid>', 'Mark messages as seen via FB Messenger bot')
  .command('send-message <psid> <message>', 'Send simple message via FB Messenger bot')
  .command('typing-indicator-on <psid>', 'Enable typing indicator via FB Messenger bot')
  .command('typing-indicator-off <psid>', 'Disable typing indicator via FB Messenger bot')
  .command('heartbeat', 'Notify FB Messenger bot that someone is listening')
  .parse(process.argv);
