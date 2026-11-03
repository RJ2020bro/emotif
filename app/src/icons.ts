// Stroke icons drawn for Emotif (24-unit grid, currentColor).
const wrap = (d: string) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;

export const ICONS = {
  logo: `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><circle cx="16" cy="16" r="13"/><path d="M6.5 17h4l2.2-6 3.4 11 2.6-8 1.6 3h5.2"/></svg>`,
  live: wrap('<path d="M3 12h3.5l2-5 3.5 10 2.5-7 1.5 2H21"/>'),
  study: wrap('<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4.5h6M8.5 10h7M8.5 14h7M8.5 18h4"/>'),
  arm: wrap('<path d="M4 20h9"/><path d="M8.5 20v-3.5"/><circle cx="8.5" cy="15" r="1.6"/><path d="M9.6 13.9 14 7.5"/><circle cx="14.8" cy="6.5" r="1.4"/><path d="M16 7.2l3.2 3.3"/><path d="M18.4 12.2l1.6-1.6M19.2 9.6l1.8.6"/>'),
  models: wrap('<path d="m12 3 8.5 4.5L12 12 3.5 7.5z"/><path d="m3.5 12 8.5 4.5 8.5-4.5"/><path d="m3.5 16.5 8.5 4.5 8.5-4.5"/>'),
  sessions: wrap('<ellipse cx="12" cy="6" rx="7.5" ry="2.8"/><path d="M4.5 6v6c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8V6"/><path d="M4.5 12v6c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8v-6"/>'),
  method: wrap('<rect x="3" y="4" width="6" height="5" rx="1.2"/><rect x="15" y="4" width="6" height="5" rx="1.2"/><rect x="9" y="15" width="6" height="5" rx="1.2"/><path d="M9 6.5h6M18 9v3H12v3M6 9v3h6"/>'),
  setup: wrap('<path d="M5 4v16M12 4v16M19 4v16"/><rect x="3" y="7" width="4" height="3" rx="1"/><rect x="10" y="13" width="4" height="3" rx="1"/><rect x="17" y="9" width="4" height="3" rx="1"/>'),
  sun: wrap('<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>'),
  moon: wrap('<path d="M19.5 14.5A8 8 0 0 1 9.5 4.5a8 8 0 1 0 10 10z"/>'),
};
