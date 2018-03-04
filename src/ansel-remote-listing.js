#!/usr/bin/env node
/** @flow */
const program = require('commander');
const FlashAir = require('./flashair');

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

if (typeof hostValue === 'undefined') {
  console.error('Error: no host specified');
  process.exit(1);
}

if (typeof directoryValue === 'undefined') {
  console.error('Error: no remote directory specified');
  process.exit(1);
}

FlashAir.list(hostValue, directoryValue, (err, result) => {
  if (err) {
    process.exit(err);
  }
  const listing = FlashAir.fileListToListing(result);

  console.log(listing);
});
