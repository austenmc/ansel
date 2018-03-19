/** @flow */

import { filesToSync } from '../src/syncFiles';

describe('syncFiles', () => {
  it('returns all remote on empty local file list', () => {
    const local = {
      Destination: {
        type: 'directory',
        name: 'Destination',
        path: '/Destination',
        contents: {},
      },
    };

    const remoteContents = {
      'file.2': {
        directory: '/DCIM',
        modified: new Date('2018-01-01T06:05:02'),
        name: 'file.2',
        path: '/DCIM/file.2',
        size: 7,
        type: 'file',
      },
      'file.1': {
        directory: '/DCIM',
        modified: new Date('2018-01-01T06:05:02'),
        name: 'file.1',
        path: '/DCIM/file.1',
        size: 7,
        type: 'file',
      },
    };

    const remote = {
      DCIM: {
        type: 'directory',
        name: 'DCIM',
        path: '/DCIM',
        contents: remoteContents,
      },
    };

    expect(filesToSync({ remote, local })).toEqual(remoteContents);
  });

  it('returns only remotes with different size', () => {
    const local = {
      Destination: {
        type: 'directory',
        name: 'Destination',
        path: '/Destination',
        contents: {
          'Photos-2018-01': {
            type: 'directory',
            name: 'Photos-2018-01',
            path: '/Photos-2018-01',
            contents: {
              'file.3': {
                directory: '/Photos-2018-01',
                modified: new Date('2018-01-01T06:05:02'),
                name: 'file.3',
                path: '/Photos-2018-01/file.4',
                size: 5,
                type: 'file',
              },
              'file.4': {
                directory: '/Photos-2018-01',
                modified: new Date('2018-01-01T06:05:02'),
                name: 'file.4',
                path: '/Photos-2018-01/file.4',
                size: 7,
                type: 'file',
              },
            },
          },
        },
      },
    };

    const remote = {
      DCIM: {
        type: 'directory',
        name: 'DCIM',
        path: '/DCIM',
        contents: {
          'file.3': {
            directory: '/DCIM',
            modified: new Date('2018-01-01T06:05:02'),
            name: 'file.3',
            path: '/DCIM/file.3',
            size: 7,
            type: 'file',
          },
          'file.4': {
            directory: '/DCIM',
            modified: new Date('2018-01-01T06:05:02'),
            name: 'file.4',
            path: '/DCIM/file.4',
            size: 7,
            type: 'file',
          },
        },
      },
    };

    expect(filesToSync({ remote, local })).toEqual({
      'file.3': {
        directory: '/DCIM',
        modified: new Date('2018-01-01T06:05:02'),
        name: 'file.3',
        path: '/DCIM/file.3',
        size: 7,
        type: 'file',
      },
    });
  });
});
