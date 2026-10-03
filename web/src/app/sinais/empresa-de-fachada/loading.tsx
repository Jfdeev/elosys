import { Skeleton } from "@/components/skeleton";

export default function Loading() {
  return (
    <div className="flex flex-col gap-8">
      <section>
        <Skeleton className="h-3 w-80" />
        <Skeleton className="mt-3 h-7 w-96" />
        <Skeleton className="mt-4 h-4 w-full max-w-2xl" />
        <Skeleton className="mt-1.5 h-4 w-2/3 max-w-2xl" />
        <div className="mt-6 flex flex-wrap items-center gap-2">
          <Skeleton className="h-8 w-40" style={{ borderRadius: "var(--r-md)" }} />
          <Skeleton className="h-8 w-24" style={{ borderRadius: "var(--r-md)" }} />
        </div>
      </section>

      <section className="table-wrap">
        <div className="p-4">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="flex items-center gap-4 border-b border-[var(--border-1)] py-3 last:border-0">
              <Skeleton className="h-3.5 flex-1" />
              <Skeleton className="h-3.5 w-32" />
              <Skeleton className="h-3.5 w-24" />
              <Skeleton className="h-3.5 w-20" />
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
