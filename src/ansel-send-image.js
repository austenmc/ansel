#!/usr/bin/env node
/** @flow */
const _ = require('lodash');
const program = require('commander');
const sendImage = require('./sendImage');

let psidValue = '';
let tokenValue = '';
let fileValue = '';

program
  .name('send-image')
  .description('Send an image file as a message attachment via FB Messenger bot.')
  .arguments('<psid> <token> <file>')
  .action((psid, token, file) => {
    psidValue = psid || '';
    tokenValue = token || '';
    fileValue = file || '';
  })
  .parse(process.argv);

if (psidValue === '') {
  console.error('Error: no psid specified');
  process.exit(1);
}

if (tokenValue === '') {
  console.error('Error: no token specified');
  process.exit(1);
}

if (fileValue === '') {
  console.error('Error: no file specified');
  process.exit(1);
}

sendImage(psidValue, tokenValue, fileValue, (err, httpResponse, jsonBody) => {
  if (err) {
    console.error('Error: ', err);
    process.exit(1);
  }

  if (_.has(jsonBody, 'error')) {
    console.error('Error: ', jsonBody.error.message);
    process.exit(1);
  }
});
