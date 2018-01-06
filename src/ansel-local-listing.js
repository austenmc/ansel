#!/usr/bin/env node
/** @flow */
const program = require('commander');
const localListing = require('./localListing');

let directoryValue = '';

program
  .name('local-listing')
  .description('List contents of local directory.')
  .arguments('<directory>')
  .action((directory) => {
    directoryValue = directory || '';
  })
  .parse(process.argv);

if (directoryValue === '') {
  console.error('Error: no local directory specified');
  process.exit(1);
}

const output = localListing(directoryValue);
console.log(JSON.stringify(output));
