/** @flow */

import paths from '../src/paths';

describe('paths', () => {
  it('handles an empty filelisting', () => {
    expect(paths({})).toBe('');
  });

  it('returns simple, nested list without status', () => {
    expect(paths({
      'foo.bar': {
        type: 'file',
        name: 'foo.bar',
        path: 'path/to/foo.bar',
        size: 13,
        modified: new Date('2018-01-01'),
      },
      'test.file': {
        type: 'file',
        name: 'test.file',
        path: 'path/to/test.file',
        size: 13,
        modified: new Date('2018-01-01'),
      },
      directory: {
        type: 'directory',
        name: 'directory',
        path: 'path/to/directory',
        contents: {
          'bar.baz': {
            type: 'file',
            name: 'bar.baz',
            path: 'path/to/directory/bar.baz',
            size: 13,
            modified: new Date('2018-01-01'),
          },
        },
      },
    })).toBe('path/to/foo.bar\npath/to/test.file\npath/to/directory/bar.baz\n');
  });

  it('returns simple, nested list with all status', () => {
    expect(paths({
      'foo.bar': {
        type: 'file',
        name: 'foo.bar',
        path: 'path/to/foo.bar',
        size: 13,
        status: 'ok',
        modified: new Date('2018-01-01'),
      },
      'test.file': {
        type: 'file',
        name: 'test.file',
        path: 'path/to/test.file',
        size: 13,
        status: 'failed',
        modified: new Date('2018-01-01'),
      },
      directory: {
        type: 'directory',
        name: 'directory',
        path: 'path/to/directory',
        contents: {
          'bar.baz': {
            type: 'file',
            name: 'bar.baz',
            path: 'path/to/directory/bar.baz',
            size: 13,
            status: 'ok',
            modified: new Date('2018-01-01'),
          },
        },
      },
    })).toBe('path/to/foo.bar\npath/to/test.file\npath/to/directory/bar.baz\n');
  });

  it('returns nested list with certain status', () => {
    expect(paths({
      'foo.bar': {
        type: 'file',
        name: 'foo.bar',
        path: 'path/to/foo.bar',
        size: 13,
        status: 'ok',
        modified: new Date('2018-01-01'),
      },
      'test.file': {
        type: 'file',
        name: 'test.file',
        path: 'path/to/test.file',
        size: 13,
        status: 'failed',
        modified: new Date('2018-01-01'),
      },
      directory: {
        type: 'directory',
        name: 'directory',
        path: 'path/to/directory',
        contents: {
          'bar.baz': {
            type: 'file',
            name: 'bar.baz',
            path: 'path/to/directory/bar.baz',
            size: 13,
            status: 'ok',
            modified: new Date('2018-01-01'),
          },
        },
      },
    }, 'ok')).toBe('path/to/foo.bar\npath/to/directory/bar.baz\n');
  });
});
