/**
 * Minimal IndexedDB wrapper for session history.
 *
 * Deliberately dependency-free: this stores a few hundred small records and
 * needs one index, which does not justify pulling in a wrapper library.
 *
 * Everything stays on the user's device. Nothing here talks to a network, and
 * no video or landmark data is retained — only the per-session summary the
 * results screen already shows.
 */

import type { SessionRecord } from './types';

const DB_NAME = 'mavis';
const DB_VERSION = 1;
const STORE = 'sessions';
const DATE_INDEX = 'by-date';

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' });
        // Sorted reads for the history list and trend chart.
        store.createIndex(DATE_INDEX, 'date');
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  // Don't cache a rejected promise — a transient failure shouldn't disable
  // storage for the rest of the page's life.
  dbPromise.catch(() => {
    dbPromise = null;
  });

  return dbPromise;
}

function tx<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const transaction = db.transaction(STORE, mode);
        const request = run(transaction.objectStore(STORE));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      }),
  );
}

/** True when the browser can persist history at all (e.g. not private mode). */
export function isAvailable(): boolean {
  try {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  } catch {
    return false;
  }
}

export async function saveSession(record: SessionRecord): Promise<void> {
  await tx('readwrite', (store) => store.put(record));
}

/** All sessions, newest first. */
export async function listSessions(): Promise<SessionRecord[]> {
  const all = await tx<SessionRecord[]>('readonly', (store) => store.getAll());
  return all.sort((a, b) => b.date.localeCompare(a.date));
}

export async function getSession(id: string): Promise<SessionRecord | undefined> {
  return tx<SessionRecord | undefined>('readonly', (store) => store.get(id));
}

export async function deleteSession(id: string): Promise<void> {
  await tx('readwrite', (store) => store.delete(id));
}

export async function clearAll(): Promise<void> {
  await tx('readwrite', (store) => store.clear());
}

/** Test seam — drops the cached connection so a fresh DB can be opened. */
export function _resetConnectionForTests(): void {
  dbPromise = null;
}
