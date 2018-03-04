/** @flow */

import localListing from '../src/localListing';

describe('localListing', () => {
  it('handles a non-existent directory', () => {
    expect(localListing(`${__dirname}/cases/does-not-exist`)).toEqual({});
  });

  it('handles an empty directory', () => {
    expect(localListing(`${__dirname}/cases/empty`)).toEqual({
      empty: {
        contents: {},
        directory: `${__dirname}/cases`,
        name: 'empty',
        path: `${__dirname}/cases/empty`,
        type: 'directory',
      },
    });
  });

  it('handles simple nesting directory', () => {
    expect(localListing(`${__dirname}/cases/test1`)).toEqual({
      test1: {
        directory: `${__dirname}/cases`,
        name: 'test1',
        path: `${__dirname}/cases/test1`,
        type: 'directory',
        contents: {
          directory1: {
            contents: {
              'file.2': {
                directory: `${__dirname}/cases/test1/directory1`,
                modified: new Date('2018-01-15T22:35:18.000'),
                name: 'file.2',
                path: `${__dirname}/cases/test1/directory1/file.2`,
                size: 8,
                type: 'file',
              },
            },
            directory: `${__dirname}/cases/test1`,
            name: 'directory1',
            path: `${__dirname}/cases/test1/directory1`,
            type: 'directory',
          },
          'file.1': {
            directory: `${__dirname}/cases/test1`,
            modified: new Date('2018-01-15T22:35:10.000'),
            name: 'file.1',
            path: `${__dirname}/cases/test1/file.1`,
            size: 7,
            type: 'file',
          },
        },
      },
    });
  });
});
