/** @flow */

import localListing from '../src/localListing';

describe('localListing', () => {
  it('handles a non-existent directory', () => {
    expect(localListing(`${__dirname}/cases/does-not-exist`)).toEqual({});
  });

  it('handles an empty directory', () => {
    expect(localListing(`${__dirname}/cases/empty`)).toEqual({});
  });

  it('handles simple nesting directory', () => {
    expect(localListing(`${__dirname}/cases/test1`)).toEqual({
      directory1: {
        contents: {
          'file.2': {
            directory: '/Users/austenmc/ansel/__tests__/cases/test1/directory1',
            modified: new Date('2018-01-15T22:35:18.904Z'),
            name: 'file.2',
            path: '/Users/austenmc/ansel/__tests__/cases/test1/directory1/file.2',
            size: 8,
            type: 'file',
          },
        },
        directory: '/Users/austenmc/ansel/__tests__/cases/test1',
        name: 'directory1',
        path: '/Users/austenmc/ansel/__tests__/cases/test1/directory1',
        type: 'directory',
      },
      'file.1': {
        directory: '/Users/austenmc/ansel/__tests__/cases/test1',
        modified: new Date('2018-01-15T22:35:10.160Z'),
        name: 'file.1',
        path: '/Users/austenmc/ansel/__tests__/cases/test1/file.1',
        size: 7,
        type: 'file',
      },
    });
  });
});
