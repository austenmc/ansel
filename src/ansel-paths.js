#!/usr/bin/env node
/** @flow */
const program = require('commander');

const { stdin } = process;
const paths = require('./paths');

program
  .name('paths')
  .description('Print a simple list of paths extracted from a file list provided on stdin.')
  .option('--status <status>', 'Return only files with specified status. Try ok or failed.')
  .parse(process.argv);

const inputChunks = [];

stdin.resume();
stdin.setEncoding('utf8');

stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

stdin.on('end', () => {
  const inputJSON = inputChunks.join('');
  const parsedData = JSON.parse(inputJSON);

  const output = paths(parsedData, program.status);

  console.log(output);
});
