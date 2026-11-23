import type { ReactNode } from "react";

// A small Markdown renderer for model replies: headings, paragraphs, lists, fenced code,
// tables, bold, italics, inline code and links. It builds React elements and never sets
// HTML, so text a model wrote cannot run script in the page that holds the user's token.
export function Markdown({ text }: { text: string }) {
  return <div className="md">{blocks(text)}</div>;
}

function blocks(text: string): ReactNode[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const out: ReactNode[] = [];
  let i = 0;
  const key = () => `b${out.length}`;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    const fence = line.match(/^\s*(```+|~~~+)\s*([\w+-]*)/);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i++]);
      i++;
      out.push(
        <pre key={key()} className="md-code">
          <code>{body.join("\n")}</code>
        </pre>,
      );
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 2, 6);
      const Tag = `h${level}` as "h3";
      out.push(<Tag key={key()}>{inline(heading[2])}</Tag>);
      i++;
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      const rows: string[][] = [];
      const cells = (l: string) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(line);
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(
        <div key={key()} className="md-table">
          <table>
            <thead>
              <tr>
                {head.map((c, j) => (
                  <th key={j}>{inline(c)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, j) => (
                <tr key={j}>
                  {r.map((c, k) => (
                    <td key={k}>{inline(c)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }
    const item = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
    if (item.test(line)) {
      const ordered = /^\s*\d/.test(line);
      const items: { depth: number; text: string }[] = [];
      while (i < lines.length && (item.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        const m = lines[i].match(item);
        if (m) items.push({ depth: Math.floor(m[1].length / 2), text: m[3] });
        else items[items.length - 1].text += ` ${lines[i].trim()}`;
        i++;
      }
      out.push(list(items, ordered, key()));
      continue;
    }
    if (/^\s*>/.test(line)) {
      const quote: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(<blockquote key={key()}>{blocks(quote.join("\n"))}</blockquote>);
      continue;
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      out.push(<hr key={key()} />);
      i++;
      continue;
    }
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,6}\s|\s*```|\s*~~~|\s*>|\s*([-*+]|\d+[.)])\s)/.test(lines[i]) &&
      !/^\s*\|.*\|\s*$/.test(lines[i])
    )
      para.push(lines[i++]);
    if (!para.length) para.push(lines[i++]);
    out.push(<p key={key()}>{inline(para.join("\n"))}</p>);
  }
  return out;
}

function list(items: { depth: number; text: string }[], ordered: boolean, key: string): ReactNode {
  const Tag = ordered ? "ol" : "ul";
  const top: { text: string; children: { depth: number; text: string }[] }[] = [];
  const base = Math.min(...items.map((x) => x.depth));
  for (const it of items) {
    if (it.depth === base || !top.length) top.push({ text: it.text, children: [] });
    else top[top.length - 1].children.push(it);
  }
  return (
    <Tag key={key}>
      {top.map((t, j) => (
        <li key={j}>
          {inline(t.text)}
          {t.children.length > 0 && list(t.children, false, `${key}.${j}`)}
        </li>
      ))}
    </Tag>
  );
}

const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\s][^*]*\*)|(\[[^\]]+\]\(https?:\/\/[^)\s]+\))/;

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let rest = text;
  let n = 0;
  while (rest) {
    const m = rest.match(INLINE);
    if (!m || m.index === undefined) {
      out.push(...withBreaks(rest, n));
      break;
    }
    if (m.index) out.push(...withBreaks(rest.slice(0, m.index), n));
    const tok = m[0];
    const k = `i${n++}`;
    if (tok.startsWith("`")) out.push(<code key={k}>{tok.slice(1, -1)}</code>);
    else if (tok.startsWith("**") || tok.startsWith("__")) out.push(<strong key={k}>{inline(tok.slice(2, -2))}</strong>);
    else if (tok.startsWith("*")) out.push(<em key={k}>{inline(tok.slice(1, -1))}</em>);
    else {
      const link = tok.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
      if (link)
        out.push(
          <a key={k} href={link[2]} target="_blank" rel="noreferrer noopener">
            {link[1]}
          </a>,
        );
    }
    rest = rest.slice(m.index + tok.length);
  }
  return out;
}

function withBreaks(text: string, n: number): ReactNode[] {
  return text.split("\n").flatMap((part, j) => (j ? [<br key={`br${n}.${j}`} />, part] : [part]));
}
