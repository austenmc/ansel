#!/usr/bin/env node
/** @flow */
const _ = require('lodash');
const program = require('commander');
const request = require('request');

const { stdin } = process;

let psidValue = '';

program
  .name('delete-messages')
  .description('Mark messages as seen via FB Messenger bot')
  .arguments('<psid>')
  .action((psid) => {
    psidValue = psid || '';
  })
  .parse(process.argv);

if (psidValue === '') {
  console.error('Error: no psid specified');
  process.exit(1);
}

const url = `https://ansel.glitch.me/messages/${psidValue}`;

const inputChunks = [];

stdin.resume();
stdin.setEncoding('utf8');

stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

stdin.on('end', () => {
  const inputJSON = inputChunks.join('');
  const parsedData = JSON.parse(inputJSON);

  if (parsedData.messages.length === 0) {
    process.exit(0);
  }

  request({
    url,
    method: 'DELETE',
    body: parsedData,
    json: true,
  }, (err, httpResponse, jsonBody) => {
    if (err) {
      console.error('Error: ', err);
      process.exit(1);
    }

    if (_.has(jsonBody, 'error')) {
      console.error('Error: ', jsonBody.error.message);
      process.exit(1);
    }
  });
});
