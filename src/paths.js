/** @flow */
import type { FileStatus, Listing } from './types';

const _ = require('lodash');

function paths(listing: Listing, status: ?FileStatus): string {
  let output = '';

  _.forEach(listing, (value) => {
    if (value.type === 'directory') {
      output += paths(value.contents, status);
    } else if (value.type === 'file') {
      if (!_.has(value, 'status') || status === undefined || value.status === status) {
        output += `${value.path}\n`;
      }
    }
  });
  return output;
}

module.exports = paths;
