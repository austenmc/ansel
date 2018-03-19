#!/usr/bin/env node
/** @flow */
const _ = require('lodash');
const program = require('commander');
const request = require('request');

program
  .name('heartbeat')
  .description('Notify FB Messenger bot someone is listening')
  .parse(process.argv);

const url = 'https://ansel.glitch.me/heartbeat';

request({
  url,
  method: 'POST',
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
