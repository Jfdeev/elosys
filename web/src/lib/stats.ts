import { db } from "./db";
import {
  getCircularDonationSummary,
  getSupplierPartnerSummary,
  getAiReviewSummary,
  getDiscourseSummary,
  getDisproportionateExpenseCount,
  getShellSupplierSummary,
} from "./queries";

export type HomeStats = {
  people: number;
  candidacies: number;
  campaignOrgs: number;
  socialMedia: number;
  donationsTotalCents: number;
  expensesTotalCents: number;
  years: string;
};

// The .db is rewrite-only, so these full-table sums are cached per process.
const cache = new Map<number, HomeStats>();

export function getHomeStats(year?: number): HomeStats {
  const key = year ?? 0;
  const hit = cache.get(key);
  if (hit) return hit;

  const yearFilter = year != null ? " WHERE year = ?" : "";
  const args = year != null ? [year] : [];
  const c = (sql: string) => (db().prepare(sql).get(...args) as { n: number }).n;
  const sum = (sql: string) => (db().prepare(sql).get(...args) as { n: number }).n ?? 0;
  const years = db()
    .prepare("SELECT min(year) lo, max(year) hi FROM politician_history")
    .get() as { lo: number; hi: number };

  const result: HomeStats = {
    people:
      year != null
        ? c("SELECT count(DISTINCT person_id) n FROM politician_history WHERE year = ?")
        : c("SELECT count(*) n FROM people"),
    candidacies: c(`SELECT count(*) n FROM politician_history${yearFilter}`),
    campaignOrgs: c(`SELECT count(*) n FROM campaign_org${yearFilter}`),
    socialMedia: c(`SELECT count(*) n FROM social_media${yearFilter}`),
    donationsTotalCents: sum(`SELECT coalesce(sum(amount_cents),0) n FROM campaign_donation${yearFilter}`),
    expensesTotalCents: sum(`SELECT coalesce(sum(amount_cents),0) n FROM campaign_expense${yearFilter}`),
    years: year != null ? String(year) : years.lo && years.hi ? `${years.lo}–${years.hi}` : "—",
  };
  cache.set(key, result);
  return result;
}

export type SidebarCounts = {
  circularDonations: number;
  supplierPartner: number;
  aiReview: number;
  discourse: number;
  disproportionateExpense: number;
  shellSupplier: number;
};

let cachedCounts: SidebarCounts | null = null;

export function getSidebarCounts(): SidebarCounts {
  if (cachedCounts) return cachedCounts;
  cachedCounts = {
    circularDonations: getCircularDonationSummary().total,
    supplierPartner: getSupplierPartnerSummary().total,
    aiReview: getAiReviewSummary().total,
    discourse: getDiscourseSummary().total,
    disproportionateExpense: getDisproportionateExpenseCount(),
    shellSupplier: getShellSupplierSummary().total,
  };
  return cachedCounts;
}
