/** @flow */
const fs = require('fs');
const request = require('request');

function sendImage(psid: string, token: string, path: string, callback: Function) {
  const url = `https://graph.facebook.com/v2.6/me/messages?access_token=${token}`;
  const formData = {
    messaging_type: 'UPDATE',
    recipient: { id: psid },
    message: {
      attachment: {
        type: 'image',
        payload: {},
      },
    },
    filedata: fs.createReadStream(path),
  };

  request.post({ url, formData, json: true }, callback);
}

module.exports = sendImage;
