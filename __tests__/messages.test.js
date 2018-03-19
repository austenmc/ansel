/** @flow */

import messages from '../src/messages';

describe('messages', () => {
  it('handles an empty messages', () => {
    expect(messages({ messages: [] })).toBe('');
  });

  it('returns simple, sorted message list', () => {
    expect(messages({
      messages: [
        {
          psid: 'psid1',
          message: 'message 2',
          timestamp: 50,
        },
        {
          psid: 'psid2',
          message: 'message 1',
          timestamp: 25,
        },
      ],
    })).toBe('message 1\nmessage 2\n');
  });
});
