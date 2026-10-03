import Link from "next/link";
import {
  SHELL_SUPPLIER_TYPES,
  getShellSupplierCount,
  getShellSuppliers,
  getShellSupplierSummary,
  type ShellSupplierFilter,
  type ShellSupplierType,
} from "@/lib/queries";
import { formatBRL, formatCnpj } from "@/lib/format";
import { PageHeader } from "@/components/shell/shell-context";
import { PaginationLinks } from "@/components/ui/pagination-links";
import { EmptyState } from "@/components/ui/empty-state";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 40;
const MAX_CANDIDATES_SHOWN = 3;

const TYPE_LABEL: Record<ShellSupplierType, string> = {
  paid_before_company_opened: "pago antes da abertura",
  paid_after_company_inactive: "pago após inatividade",
  new_company_low_capital_high_value: "empresa nova, capital baixo, valor alto",
};

export default async function EmpresaDeFachadaPage({ searchParams }: PageProps<"/sinais/empresa-de-fachada">) {
  const sp = await searchParams;
  const tipoParam = typeof sp.tipo === "string" ? sp.tipo : undefined;
  const type: ShellSupplierFilter = SHELL_SUPPLIER_TYPES.find((t) => t === tipoParam) ?? "all";
  const q = typeof sp.q === "string" ? sp.q : "";
  const page = Math.max(1, Number(sp.page) || 1);

  const summary = getShellSupplierSummary();
  const count = getShellSupplierCount({ type, q });
  const rows = getShellSuppliers({ type, q, limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE });
  const totalPages = Math.max(1, Math.ceil(count / PAGE_SIZE));

  const hrefFor = (p: Record<string, string | undefined>) => {
    const usp = new URLSearchParams();
    if (p.tipo && p.tipo !== "all") usp.set("tipo", p.tipo);
    if (p.q) usp.set("q", p.q);
    if (p.page && p.page !== "1") usp.set("page", p.page);
    const s = usp.toString();
    return `/sinais/empresa-de-fachada${s ? `?${s}` : ""}`;
  };

  return (
    <div className="flex flex-col gap-8">
      <PageHeader group="Sinais" current="Empresa de fachada" />

      <section>
        <h1 className="text-[26px] leading-tight font-medium tracking-tight">
          Fornecedores cujo cadastro na Receita não combina com o que receberam de campanhas
        </h1>
        <p className="mt-4 max-w-2xl text-[13px] leading-relaxed" style={{ color: "var(--muted)" }}>
          Compara a <strong style={{ color: "var(--fg-2)" }}>data de abertura, a situação cadastral e o capital
          social</strong> de cada fornecedor com as datas e os valores dos pagamentos de campanha. Datas
          cadastrais podem estar erradas, um CNPJ pode ser reaproveitado e muitas empresas legítimas abrem
          perto da eleição ou operam com pouco capital. Só são avaliados fornecedores com dados da Receita
          já coletados (<code>receita-cnpj</code>); repasses entre campanhas e órgãos de partidos ficam de
          fora. Indício, não prova.
        </p>

        {summary.total === 0 ? null : (
          <div className="mono mt-6 flex flex-wrap items-center gap-x-6 gap-y-2" style={{ fontSize: 11, color: "var(--muted)" }}>
            <span>
              <span style={{ color: "var(--fg-1)" }}>{summary.total.toLocaleString("pt-BR")}</span> sinais
            </span>
            <span>
              <span style={{ color: "var(--red)" }}>{summary.byType.paid_before_company_opened.toLocaleString("pt-BR")}</span>{" "}
              pagos antes da abertura
            </span>
            <span>
              <span style={{ color: "var(--red)" }}>{summary.byType.paid_after_company_inactive.toLocaleString("pt-BR")}</span>{" "}
              pagos após inatividade
            </span>
            <span>
              <span style={{ color: "var(--accent-2)" }}>
                {summary.byType.new_company_low_capital_high_value.toLocaleString("pt-BR")}
              </span>{" "}
              empresas novas com capital baixo
            </span>
          </div>
        )}

        <form className="mt-5 flex flex-wrap items-center gap-2" action="/sinais/empresa-de-fachada">
          <Link href={hrefFor({ tipo: "all", q })} className={`btn${type === "all" ? " btn--primary" : ""}`}>
            todos
          </Link>
          {SHELL_SUPPLIER_TYPES.map((t) => (
            <Link key={t} href={hrefFor({ tipo: t, q })} className={`btn${type === t ? " btn--primary" : ""}`}>
              {TYPE_LABEL[t]}
            </Link>
          ))}
          <div className="input ml-2" style={{ width: 256 }}>
            <input name="q" defaultValue={q} placeholder="buscar empresa, CNPJ ou candidato…" />
          </div>
          {type !== "all" ? <input type="hidden" name="tipo" value={type} /> : null}
        </form>
      </section>

      <section>
        {summary.total === 0 ? (
          <EmptyState
            icon="◌"
            title="nenhum sinal ainda"
            hint={
              <>
                <code>elosys rule-shell-supplier --db elosys.db</code> (precisa de dados cadastrais já
                coletados via <code>receita-cnpj</code>)
              </>
            }
          />
        ) : rows.length === 0 ? (
          <EmptyState icon="◌" title="nada para esse filtro." />
        ) : (
          <div className="table-wrap">
            <div className="overflow-x-auto">
              <table className="table min-w-[860px]">
                <thead>
                  <tr>
                    <th>indício</th>
                    <th>empresa</th>
                    <th>candidato(s)</th>
                    <th className="text-right">valor</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.signalId}>
                      <td className="max-w-[420px]">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className={`badge ${r.severity === "high" ? "badge--red" : "badge--accent"}`}>
                            {r.severity}
                          </span>
                          <span className="mono" style={{ fontSize: 10, color: "var(--muted-2)" }}>
                            {TYPE_LABEL[r.type] ?? r.type}
                          </span>
                        </div>
                        <p className="mt-1.5 text-[11.5px] leading-snug" style={{ color: "var(--muted)" }}>
                          {r.explanation}
                        </p>
                      </td>
                      <td className="max-w-[240px]">
                        <Link href={`/cnpj/${r.companyCnpj}`} className="hover:underline">
                          {r.companyName ?? formatCnpj(r.companyCnpj)}
                        </Link>
                        <div className="mono" style={{ fontSize: 9.5, color: "var(--muted-2)" }}>
                          {formatCnpj(r.companyCnpj)}
                        </div>
                      </td>
                      <td className="max-w-[220px]">
                        {r.candidates.slice(0, MAX_CANDIDATES_SHOWN).map((c) => (
                          <div key={c.personId}>
                            <Link href={`/politico/${c.personId}`} className="hover:underline">
                              {c.name ?? "candidato"}
                            </Link>
                          </div>
                        ))}
                        {r.candidates.length > MAX_CANDIDATES_SHOWN ? (
                          <div className="mono" style={{ fontSize: 9.5, color: "var(--muted-2)" }}>
                            +{r.candidates.length - MAX_CANDIDATES_SHOWN} candidato(s)
                          </div>
                        ) : null}
                      </td>
                      <td className="num" style={{ color: "var(--accent-2)" }}>
                        {formatBRL(r.amountCents)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {totalPages > 1 ? (
              <div className="table-footer">
                <PaginationLinks
                  page={page}
                  totalPages={totalPages}
                  makeHref={(p) => hrefFor({ tipo: type, q, page: String(p) })}
                />
              </div>
            ) : null}
          </div>
        )}
      </section>
    </div>
  );
}
