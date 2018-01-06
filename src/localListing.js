/** @flow */
import type { FileListing } from './types';

const fs = require('fs');
const path = require('path');

function walk(dir: string): FileListing {
  const o: FileListing = {};

  fs.readdirSync(dir).filter((f) => f && f[0] !== '.') // Ignore hidden files
    .forEach((f) => {
      const p = path.join(dir, f);
      const stat = fs.statSync(p);

      if (stat.isDirectory()) {
        o[f] = {
          type: 'directory',
          name: f,
          path: p,
          directory: dir,
          contents: walk(p),
        };
      } else {
        o[f] = {
          type: 'file',
          name: f,
          directory: dir,
          path: p,
          size: stat.size,
          modified: stat.mtime,
        };
      }
    });

  return o;
}

function localListing(directory: string): FileListing {
  let output = {};

  if (fs.existsSync(directory)) {
    output = walk(directory);
  }

  return output;
}

module.exports = localListing;
