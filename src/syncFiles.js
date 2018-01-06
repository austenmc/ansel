/** @flow */
import type { File, Directory, FileListing, SyncListing } from './types';

const _ = require('lodash');
const dateFns = require('date-fns');
// const FlashAir = require('./flashair');

function directoryFromDate(mtime: Date): string {
  return `RAW-${dateFns.format(mtime, 'YYYY-MM')}`;
}

// Uses size to compare the files, so is vulnerable to changing file contents.
function hasUpToDateLocal(file: File, local: FileListing): boolean {
  const dir = directoryFromDate(file.modified);
  if (_.has(local, dir) && local[dir].type === 'directory') {
    const contents = (local[dir]: Directory);
    if (_.has(contents, file.name)) {
      if (contents[file.name].size === file.size) {
        return true;
      }
    }
  }
  return false;
}

function syncFiles(listing: SyncListing): FileListing {
  const output = {};

  _.forEach(listing.remote, (file: File, name: string) => {
    if (!hasUpToDateLocal(file, listing.local)) {
      output[name] = file;

      //      FlashAir.download();

      output[name].status = 'OK';
    }
  });

  return output;
}

module.exports = syncFiles;
