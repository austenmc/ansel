#!/usr/bin/env node
/** @flow */
const program = require('commander');
const FlashAir = require('./flashair');
const path = require('path');

let hostValue = '';
let directoryValue = '';

program
  .name('remote-listing')
  .description('List contents of remote directory')
  .arguments('<host> <directory>')
  .action((host, directory) => {
    hostValue = host;
    directoryValue = directory;
  })
  .parse(process.argv);

if (hostValue.length == 0) {
  console.error('Error: no host specified');
  process.exit(1);
}

if (directoryValue.length == 0) {
  console.error('Error: no remote directory specified');
  process.exit(1);
}

FlashAir.list(hostValue, directoryValue, (err, result) => {
  if (err) {
    process.exit(err);
  }
  const listing = FlashAir.fileListToListing(result);

  const name = path.basename(directoryValue);
  const parent = path.dirname(directoryValue);
  const output = {
    [directoryValue]: {
      type: 'directory',
      name,
      directory: parent,
      path: directoryValue,
      contents: listing,
    }
  };

  console.log(JSON.stringify(output));
});
