#!/usr/bin/env node
/** @flow */
const _ = require('lodash');
const program = require('commander');
const request = require('request');

let psidValue = '';

program
  .name('typing-indicator-on')
  .description('Enable typing indicator via FB Messenger bot')
  .arguments('<psid>')
  .action((psid) => {
    psidValue = psid || '';
  })
  .parse(process.argv);

if (psidValue === '') {
  console.error('Error: no psid specified');
  process.exit(1);
}

const url = `https://ansel.glitch.me/typing-indicator/${psidValue}`;

request({
  url,
  method: 'PUT',
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
