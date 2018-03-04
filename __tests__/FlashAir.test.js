/** @flow */

import FlashAir from '../src/flashair';

describe('FlashAir', () => {
  it('parses file type=directory bits', () => {
    expect(FlashAir.fileTypeFromBits(`${0b1110000}`)).toEqual('directory');
  });
  it('parses file type=file bits', () => {
    expect(FlashAir.fileTypeFromBits(`${0b1010000}`)).toEqual('file');
  });

  it('parses date bits', () => {
    expect(FlashAir.dateFromDateTimeBits(`${0b0100110000100001}`, `${0b0011000010100001}`)).toEqual(new Date('2018-01-01T06:05:02'));
  });

  it('converts command.cgi output to FileListing', () => {
    const filelist =
`WLANSD_FILELIST
/DCIM,file.1,7,${0b1010000},${0b0100110000100001},${0b0011000010100001}
/DCIM,file.2,7,${0b1010000},${0b0100110000100001},${0b0011000010100001}
/DCIM,directory.1,7,${0b1110000},${0b0100110000100001},${0b0011000010100001}
`;
    expect(FlashAir.fileListToListing(filelist)).toEqual({
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
    });
  });
});
