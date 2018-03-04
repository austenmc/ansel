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

export type DirectoryListing = {
  [string]: Directory
};

export type SyncListing = {
  remote: DirectoryListing,
  local: DirectoryListing,
};

/* eslint-enable no-use-before-define */
