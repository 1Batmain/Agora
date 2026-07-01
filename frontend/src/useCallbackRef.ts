import { useCallback, useEffect, useRef } from 'react';

/**
 * Returns a stable function identity that always invokes the latest `fn`.
 * Lets a debounce timer call the current closure without re-arming on re-render.
 */
export function useCallbackRef<A extends any[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useRef(fn);
  useEffect(() => {
    ref.current = fn;
  });
  return useCallback((...args: A) => ref.current(...args), []);
}
