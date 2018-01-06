/** @flow */

/* eslint-disable no-use-before-define */

export type FileType = 'directory' | 'file';

export type FileStatus = 'ok' | 'failed';

export type Directory = {
  type: 'directory',
  name: string,
  path: string,
  contents: FileListing,
};

export type File = {
  type: 'file',
  name: string,
  path: string,
  size: number,
  modified: Date,
  status?: string,
};

export type FileListing = {
  [string]: File | Directory
};

export type SyncListing = {
  remote: FileListing,
  local: FileListing,
};

/* eslint-enable no-use-before-define */
