#!/usr/bin/env node
/** @flow */
const _ = require('lodash');
const program = require('commander');
const sendMessage = require('./sendMessage');

let psidValue = '';
let messageValue = '';

program
  .name('send-message')
  .description('Send simple message via the Ansel FB Messenger bot.')
  .arguments('<psid> <message>')
  .action((psid, message) => {
    psidValue = psid || '';
    messageValue = message || '';
  })
  .parse(process.argv);

if (psidValue === '') {
  console.error('Error: no psid specified');
  process.exit(1);
}

if (messageValue === '') {
  console.error('Error: no message specified');
  process.exit(1);
}

sendMessage(psidValue, messageValue, (err, httpResponse, jsonBody) => {
  if (err) {
    console.error('Error: ', err);
    process.exit(1);
  }

  if (_.has(jsonBody, 'error')) {
    console.error('Error: ', jsonBody.error.message);
    process.exit(1);
  }
});
