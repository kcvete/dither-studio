/* ES module face of dither-player.js.
 *
 * The player itself is a classic script so a page can drop it in with a plain
 * <script src>. A classic script cannot carry `export` statements, so this file
 * loads it for its side effect and re-exports the global it publishes. Both
 * halves are the same object — there is one implementation.
 *
 *   import { Player, buildTransition } from './dither-player.mjs';
 */
import './dither-player.js';

const P = globalThis.DitherPlayer;

export const {
  Player, encode, decode, gzip, gunzip, pack, unpack, toJSON, fromJSON,
  paintFrame, buildMorph, morphPairs, scatterPairs, densityPairs, regrid,
  buildTransition, TRANSITIONS, buildSequence, hilbertOrder, hexRGB,
  thinCloud, PARTICLE_CAP, easeInOut, mulberry32, version,
} = P;

export default P;
