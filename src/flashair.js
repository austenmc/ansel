/** @flow */
const fs = require('fs');
const request = require('request');

const FlashAir = {};

FlashAir.list = (host: string, path: string, callback: Function) => {
  const url = `${host}/command.cgi?op=100&DIR=${encodeURIComponent(path)}`;
  request.get({ url }, (err, httpResponse) => {
    let result = '';

    if (!err) {
      result = httpResponse;
    }

    callback(err, result);
  });
};

FlashAir.download = (source: string, destination: string): Promise<*> => {
  const p = new Promise((resolve, reject) => {
    const stream = fs.createWriteStream(destination);
    stream.on('error', (err) => {
      reject(err);
    });

    request.get({ uri: source }, (err) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    }).pipe(stream);
  });

  return p;
};

function pad(num, size): string {
  const s = `000000000${num}`;
  return s.substr(s.length - size);
}

FlashAir.dateFromDateTimeBits = (date: string, time: string) => {
  const t = parseInt(time, 10);
  const d = parseInt(date, 10);

  const day = d & 0b0000000000011111;
  const month = (d & 0b0000000111100000) >> 5;
  const year = ((d & 0b1111111000000000) >> 9) + 1980;

  const second = (t & 0b0000000000011111) * 2;
  const minute = (t & 0b0000011111100000) >> 5;
  const hour = (t & 0b1111100000000000) >> 11;

  const dateString = `${pad(year, 4)}-${pad(month, 2)}-${pad(day, 2)}T${pad(hour, 2)}:${pad(minute, 2)}:${pad(second, 2)}`;
  return new Date(dateString);
};

FlashAir.fileTypeFromBits = (type: string) => (parseInt(type, 10) & 0b100000 ? 'directory' : 'file');

FlashAir.fileListToListing = (fileList: string) => {
  const lines = fileList.split('\n');

  // Remove the first line, which is name of the device?
  lines.splice(0, 1);

  // Generate a listing object based on resulting file list.
  return lines.reduce((listing, l) => {
    if (l.length === 0) {
      return listing;
    }

    const tokens = l.split(',');

    const time = tokens.pop();
    const date = tokens.pop();
    const modified = FlashAir.dateFromDateTimeBits(date, time);
    const type = FlashAir.fileTypeFromBits(tokens.pop());
    const size = parseInt(tokens.pop(), 10);
    const directory = tokens.shift(); // this does not handle case where dir name contains ,
    const name = tokens.join(','); // to handle case where filename contains ,

    if (type === 'directory') {
      // We don't handle directories in the listings.
      return listing;
    }

    const o = listing;

    o[name] = {
      type,
      name,
      directory,
      size,
      path: `${directory}/${name}`,
      modified,
    };
    return o;
  }, {});
};

module.exports = FlashAir;
