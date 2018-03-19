/** @flow */
import type { File, Directory, FileListing, SyncListing } from './types';

const _ = require('lodash');
const dateFns = require('date-fns');
const fs = require('fs');
const FlashAir = require('./flashair');
const path = require('path');

function directoryFromDate(mtime: Date): string {
  return `Photos-${dateFns.format(mtime, 'YYYY-MM')}`;
}

// Uses size to compare the files, so is vulnerable to changing file contents.
function hasUpToDateLocal(file: File, local: Directory): boolean {
  const dir = directoryFromDate(file.modified);

  if (_.has(local.contents, dir) && local.contents[dir].type === 'directory') {
    const { contents } = (local.contents[dir]: Directory);
    if (_.has(contents, file.name)) {
      if (contents[file.name].size === file.size) {
        return true;
      }
    }
  }
  return false;
}

// Return a flat file listing, no directories, of remote files
// that need to be synced.
export function filesToSync(listing: SyncListing): FileListing {
  const localKeys = _.keys(listing.local);
  const localDir = listing.local[localKeys[0]];
  const remoteKeys = _.keys(listing.remote);
  const remoteDir = listing.remote[remoteKeys[0]];
  const output = {};

  _.forEach(remoteDir.contents, (file: File, name: string) => {
    if (!hasUpToDateLocal(file, localDir)) {
      output[name] = file;
    }
  });

  return output;
}

export async function syncFiles(host: string, listing: SyncListing): FileListing {
  const localKeys = _.keys(listing.local);
  if (localKeys.length < 1) {
    console.error('Error: No local destination directory specified.');
    process.exit(1);
  }
  if (localKeys.length > 1) {
    console.error('Error: Too many destination directories specified.');
    process.exit(1);
  }
  const localDir = listing.local[localKeys[0]];

  const remoteKeys = _.keys(listing.remote);
  if (remoteKeys.length < 1) {
    console.error('Error: No remote source directory specified.');
    process.exit(1);
  }
  if (remoteKeys.length > 1) {
    console.error('Error: Too many remote source directories specified.');
    process.exit(1);
  }

  const files = filesToSync(listing);
  const output = {};

  Object.keys(files).reduce(async (prev, name) => {
    await prev;

    const file = (files[name]: File);
    const dir = path.join(localDir.path, directoryFromDate(file.modified));
    const dest = path.join(dir, name);

    try {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
      }

      await FlashAir.download(`${host}/${file.path}`, dest);

      // check that the local file estsis
      fs.stat(dest, (err, stats) => {
        if (err) {
          throw new Error(err);
        }
        if (stats.size !== file.size) {
          throw new Error(`Error: ${name} downloaded but wrong size`);
        }
      });
      output[name] = { ...file, status: 'ok' };
    } catch (e) {
      output[name] = { ...file, status: 'failed' };
    }
  }, Promise.resolve());

  return output;
}
