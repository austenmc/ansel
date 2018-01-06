/** @flow */
const request = require('request');

function sendMessage(psid: string, token: string, message: string, callback: Function) {
  const url = `https://graph.facebook.com/v2.6/me/messages?access_token=${token}`;
  const data = {
    messaging_type: 'UPDATE',
    recipient: { id: psid },
    message: {
      text: message,
    },
  };

  request({
    url,
    method: 'POST',
    body: data,
    json: true,
  }, callback);
}

module.exports = sendMessage;
