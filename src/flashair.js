/** @flow */
const request = require('request');

const FlashAir = {};

FlashAir.list = function (host: string, path: string, callback: Function) {
  const url = '${host}/command.cgi?op=100&DIR=${encodeURIComponent(path)}';
  request.get({ url }, (err, httpResponse, jsonBody) => {
    let result = '';

    if (!err) {
      result = httpResponse;
    }

    callback(err, result);
  });
};

module.exports = FlashAir;
