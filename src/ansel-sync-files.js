#!/usr/bin/env node
/** @flow */
const _ = require('lodash');
const program = require('commander');
const { syncFiles } = require('./syncFiles');

const { stdin } = process;

let hostValue = '';

program
  .name('sync-files')
  .description('Sync files between a local file listing and a remote file listing. Reads JSON\n  listings from stdin in the format:\n    { remote: {...}, local: {...} }\n  See remote-listing and local-listing. Outputs a file listing, with status fields.')
  .arguments('<host>')
  .action((host) => {
    hostValue = host;
  })
  .parse(process.argv);

if (hostValue.length === 0) {
  console.error('Error: no host specified');
  process.exit(1);
}

const inputChunks = [];

stdin.resume();
stdin.setEncoding('utf8');

stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

stdin.on('end', () => {
  const inputJSON = inputChunks.join('');
  const parsedData = JSON.parse(inputJSON);

  if (!_.isObject(parsedData.remote)) {
    console.error("Error: no 'remote' object in input");
    process.exit(1);
  }
  if (!_.isObject(parsedData.local)) {
    console.error("Error: no 'local' object in input");
    process.exit(1);
  }

  const output = syncFiles(hostValue, parsedData);
  console.log(JSON.stringify(output));
});
