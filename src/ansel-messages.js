#!/usr/bin/env node
/** @flow */
const program = require('commander');
const _ = require('lodash');

const { stdin } = process;
const messages = require('./messages');

program
  .name('messages')
  .description('Extract messages text from a messages JSON listing')
  .parse(process.argv);

const inputChunks = [];

stdin.resume();
stdin.setEncoding('utf8');

stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

stdin.on('end', () => {
  const inputJSON = _.trim(inputChunks.join(''));
  if (inputJSON.length > 0) {
    const parsedData = JSON.parse(inputJSON);

    const output = messages(parsedData);

    console.log(_.trim(output));
  }
});
