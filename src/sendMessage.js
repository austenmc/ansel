/** @flow */
const request = require('request');

function sendMessage(psid: string, message: string, callback: Function) {
  const url = `https://ansel.glitch.me/messages/${psid}`;
  const data = {
    message,
  };

  request({
    url,
    method: 'POST',
    body: data,
    json: true,
  }, callback);
}

module.exports = sendMessage;
