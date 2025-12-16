/**
 * Minimal ZIP (store-only) builder for bundling multiple files client-side.
 * Supports UTF-8 filenames, no compression.
 *
 * Each file: { name: string, data: Uint8Array, date?: Date }
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : (c >>> 1);
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(data) {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = CRC_TABLE[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (~crc) >>> 0;
}

function toDosTime(date) {
  const d = date ? new Date(date) : new Date();
  const year = d.getFullYear();
  const month = d.getMonth() + 1;
  const day = d.getDate();
  const hours = d.getHours();
  const minutes = d.getMinutes();
  const seconds = Math.floor(d.getSeconds() / 2); // DOS stores seconds/2
  const dosTime = (hours << 11) | (minutes << 5) | seconds;
  const dosDate = ((year - 1980) << 9) | (month << 5) | day;
  return { dosTime, dosDate };
}

function writeUint32(view, offset, value) {
  view.setUint32(offset, value >>> 0, true);
}
function writeUint16(view, offset, value) {
  view.setUint16(offset, value & 0xffff, true);
}

/**
 * Create a ZIP blob (store method) from files.
 * @param {{name:string, data:Uint8Array, date?:Date}[]} files
 * @returns {Blob}
 */
export function createZipFromFiles(files) {
  const textEncoder = new TextEncoder();
  const localFileHeaders = [];
  const centralDirectory = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = textEncoder.encode(file.name);
    const crc = crc32(file.data);
    const { dosTime, dosDate } = toDosTime(file.date);
    const localHeaderSize = 30 + nameBytes.length;
    const localHeader = new Uint8Array(localHeaderSize);
    const view = new DataView(localHeader.buffer);

    writeUint32(view, 0, 0x04034b50); // local file header signature
    writeUint16(view, 4, 20); // version needed
    writeUint16(view, 6, 0); // flags
    writeUint16(view, 8, 0); // compression (store)
    writeUint16(view, 10, dosTime);
    writeUint16(view, 12, dosDate);
    writeUint32(view, 14, crc);
    writeUint32(view, 18, file.data.length);
    writeUint32(view, 22, file.data.length);
    writeUint16(view, 26, nameBytes.length);
    writeUint16(view, 28, 0); // extra length
    localHeader.set(nameBytes, 30);

    localFileHeaders.push(localHeader, file.data);

    const centralSize = 46 + nameBytes.length;
    const central = new Uint8Array(centralSize);
    const cv = new DataView(central.buffer);
    writeUint32(cv, 0, 0x02014b50); // central dir signature
    writeUint16(cv, 4, 20); // version made by
    writeUint16(cv, 6, 20); // version needed
    writeUint16(cv, 8, 0); // flags
    writeUint16(cv, 10, 0); // compression
    writeUint16(cv, 12, dosTime);
    writeUint16(cv, 14, dosDate);
    writeUint32(cv, 16, crc);
    writeUint32(cv, 20, file.data.length);
    writeUint32(cv, 24, file.data.length);
    writeUint16(cv, 28, nameBytes.length);
    writeUint16(cv, 30, 0); // extra len
    writeUint16(cv, 32, 0); // comment len
    writeUint16(cv, 34, 0); // disk start
    writeUint16(cv, 36, 0); // internal attrs
    writeUint32(cv, 38, 0); // external attrs
    writeUint32(cv, 42, offset); // local header offset
    central.set(nameBytes, 46);
    centralDirectory.push(central);

    offset += localHeaderSize + file.data.length;
  }

  const centralSizeTotal = centralDirectory.reduce((sum, arr) => sum + arr.length, 0);
  const endRecord = new Uint8Array(22);
  const ev = new DataView(endRecord.buffer);
  writeUint32(ev, 0, 0x06054b50); // end of central dir signature
  writeUint16(ev, 4, 0); // disk number
  writeUint16(ev, 6, 0); // disk where central directory starts
  writeUint16(ev, 8, files.length); // # of central dir records on this disk
  writeUint16(ev, 10, files.length); // total # of records
  writeUint32(ev, 12, centralSizeTotal); // size of central dir
  writeUint32(ev, 16, offset); // offset of central dir
  writeUint16(ev, 20, 0); // comment length

  const blobs = [...localFileHeaders, ...centralDirectory, endRecord];
  return new Blob(blobs, { type: 'application/zip' });
}
