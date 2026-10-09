// test_timeline_layout.js — the rows of the timeline for an MPI program: ranks on top, backends in each rank.
import { describe, it, expect } from 'vitest';
import { rankCount, layoutRows, rankSegments, segmentExtent, collectiveParts, commSegments } from '../../script/timeline_layout.js';

const heights = { regionH: 0, rowH: 28, headerH: 18, transferH: 0, commH: 0, eventH: 0, axisH: 36 };

describe('rows of the timeline', () => {
    it('a trace that is not from an MPI program has one group of backend rows and no header', () => {
        const L = layoutRows({ ...heights, ranks: 0, backends: ['CPU', 'OpenCL'] });
        expect(L.groups).toHaveLength(1);
        expect(L.groups[0]).toMatchObject({ rank: null, headerH: 0, rowY: { CPU: 0, OpenCL: 28 } });
        expect(L.axisY).toBe(56);
        expect(L.height).toBe(92);
    });

    it('an MPI program has a header and a row for each backend, for each rank', () => {
        const L = layoutRows({ ...heights, ranks: 3, backends: ['OpenMP'] });
        expect(L.groups.map(g => g.rank)).toEqual([0, 1, 2]);
        expect(L.groups.map(g => g.headerY)).toEqual([0, 46, 92]);
        expect(L.groups.map(g => g.rowY.OpenMP)).toEqual([18, 64, 110]);
        expect(L.axisY).toBe(138);
    });

    it('has the rows that are not of a rank after the ranks, in the order regions, ranks, transfers, communication, events', () => {
        const L = layoutRows({ ...heights, ranks: 2, backends: ['CPU', 'OpenMP'], regionH: 36, transferH: 28, commH: 28, eventH: 28 });
        const last = L.groups[1];
        expect(L.groups[0].headerY).toBe(36);
        const end = last.rowY.OpenMP + 28;
        expect(L.transferY).toBe(end);
        expect(L.commY).toBe(end + 28);
        expect(L.eventY).toBe(end + 56);
        expect(L.axisY).toBe(end + 84);
        expect(L.height).toBe(end + 84 + 36);
    });
});

describe('what the ranks did', () => {
    const rd = [
        { rank: 0, start: 10, end: 50, work_start: 12, work_end: 40, local_elements: 4 },
        { rank: 1, start: 5, end: 52, work_start: 20, work_end: 30, local_elements: 1 },
        { rank: 2, start: 11, end: 49, work_start: null, work_end: null, local_elements: 0 },
    ];

    it('counts the ranks of the nodes with rank data', () => {
        expect(rankCount([{ data: { rank_datas: [rd] } }, { data: {} }])).toBe(3);
        expect(rankCount([{ data: { rank_datas: [null, rd.slice(0, 2)] } }])).toBe(2);
        expect(rankCount([{ data: { start: 1, end: 2 } }])).toBe(0);
    });

    it('has a segment for each rank, with the work of the rank (none for a rank that had none)', () => {
        const s = rankSegments(rd);
        expect(s.map(x => x.rank)).toEqual([0, 1, 2]);
        expect(s[2]).toMatchObject({ workStart: null, workEnd: null, localElements: 0 });
        expect(s[1]).toMatchObject({ start: 5, end: 52, workStart: 20, workEnd: 30 });
        expect(rankSegments([rd[0], null, rd[2]]).map(x => x.rank)).toEqual([0, 2]);
        expect(rankSegments(null)).toEqual([]);
    });

    it('has the time extent of the ranks, which can be outside the interval of rank 0', () => {
        expect(segmentExtent([{ data: { rank_datas: [rd] } }])).toEqual([5, 52]);
        expect(segmentExtent([{ data: {} }])).toBeNull();
    });

    it('has two parts in a collective: the arrival (waiting), and the exchange', () => {
        const p = collectiveParts({ interval: { start: 100, end: 160 }, max_wait_us: 25 });
        expect(p).toEqual({ waitStart: 100, waitEnd: 125, exchangeStart: 125, end: 160 });
        expect(collectiveParts({ interval: { start: 100, end: 110 }, max_wait_us: 50 }).waitEnd).toBe(110);   // (clamped)
        expect(collectiveParts(null)).toBeNull();
    });
});

describe('the collectives opened up', () => {
    it('give every rank a row for them, after its backend rows', () => {
        const L = layoutRows({ ...heights, ranks: 2, backends: ['OpenMP'], rankComm: true });
        expect(L.groups.map(g => g.headerY)).toEqual([0, 74]);          // (a header, a backend row and a row for the collectives: 18 + 28 + 28)
        expect(L.groups.map(g => g.rowY.OpenMP)).toEqual([18, 92]);
        expect(L.groups.map(g => g.commY)).toEqual([46, 120]);
        expect(L.axisY).toBe(148);
        // (not for a trace that is not from an MPI program, or when they are not opened up)
        expect(layoutRows({ ...heights, ranks: 0, backends: ['CPU'], rankComm: true }).groups[0].commY).toBeUndefined();
        expect(layoutRows({ ...heights, ranks: 2, backends: ['CPU'] }).groups[0].commY).toBeUndefined();
    });

    it('have what each rank did: arrive, wait for the last, exchange', () => {
        const rd = [
            { rank: 0, start: 100, end: 160, bytes_sent: 10, bytes_received: 5 },
            { rank: 1, start: 120, end: 161, bytes_sent: 5, bytes_received: 10 },
            null,
        ];
        const segs = commSegments(rd, { interval: { start: 100, end: 161 }, max_wait_us: 20 });
        expect(segs).toEqual([
            { rank: 0, start: 100, end: 160, exchangeStart: 120, bytesSent: 10, bytesReceived: 5 },
            { rank: 1, start: 120, end: 161, exchangeStart: 120, bytesSent: 5, bytesReceived: 10 },
        ]);   // (the rank that came last did not wait)
        expect(commSegments(null, null)).toEqual([]);
    });
});
