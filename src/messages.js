/** @flow */
import type { Message, Messages } from './types';

const _ = require('lodash');

function messages(m: Messages): string {
  let output = '';

  _.forEach(_.orderBy(m.messages, ['timestamp'], ['asc']), (value: Message) => {
    output += `${value.message}\n`;
  });
  return output;
}

module.exports = messages;
