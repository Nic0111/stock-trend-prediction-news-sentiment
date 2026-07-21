import type { OverviewSceneId } from "../../data/overviewJourney";

interface MiniatureArtProps {
  kind: OverviewSceneId;
  accent: string;
}

type ArtStyle = React.CSSProperties & { "--scene-accent": string };

function Island({ accent }: { accent: string }) {
  return (
    <>
      <ellipse cx="420" cy="390" rx="356" ry="52" fill="#1b2f4b" opacity="0.14" />
      <path d="M73 304c33-91 148-151 306-161 151-9 291 22 357 105l-34 81c-87 70-229 99-384 76-125-18-222-57-245-101Z" fill="#cbd8cf" />
      <path d="M86 302c48-71 159-113 300-117 141-4 269 28 320 90-58 66-202 98-354 76-135-18-231-31-266-49Z" fill="#e5eee2" />
      <path d="M135 324c83-27 188-32 286-18 72 10 139 28 193 52-76 52-211 72-338 51-64-11-117-29-141-48Z" fill="#b8ccb9" opacity="0.82" />
      <path d="M226 287c68-48 161-73 279-69 50 2 98 10 140 23" fill="none" stroke="#f6c36a" strokeWidth="9" strokeLinecap="round" opacity="0.85" />
      <circle cx="168" cy="301" r="10" fill="#8fb697" />
      <circle cx="187" cy="326" r="7" fill="#76a582" />
      <circle cx="560" cy="307" r="9" fill="#87ad93" />
      <circle cx="585" cy="329" r="7" fill="#6f9a81" />
      <path d="M105 270c55-15 111-27 165-30" fill="none" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" opacity="0.7" />
      <path d="M522 370c50-13 86-29 120-52" fill="none" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" opacity="0.7" />
      <path d="M360 379c-13-18-17-38-12-61" fill="none" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" opacity="0.82" />
      <path d="M352 317c-15-11-18-24-9-39 18 8 25 21 20 39" fill="#91b89b" />
    </>
  );
}

function ExchangeArt({ accent }: { accent: string }) {
  return (
    <>
      <path d="M180 280h180v83H180z" fill="#f8fbff" stroke="#9eafc3" strokeWidth="5" />
      <path d="M166 281 270 211l104 70Z" fill="#6d8ed8" />
      <path d="M198 294h38v47h-38zm58 0h38v47h-38zm58 0h38v47h-38z" fill={accent} opacity="0.9" />
      <path d="M220 214c0-25 23-48 51-48s51 23 51 48" fill="none" stroke="#f2ae32" strokeWidth="10" strokeLinecap="round" />
      <circle cx="270" cy="159" r="17" fill="#f2ae32" />
      <path d="M455 276c0-37 30-67 67-67s67 30 67 67v63H455z" fill="#d9e4f4" stroke="#8aa3c4" strokeWidth="5" />
      <path d="M443 276h157l-78-75Z" fill="#2f6bff" opacity="0.82" />
      <path d="M486 292h70v39h-70z" fill="#ffffff" />
      <path d="M498 310h48" stroke={accent} strokeWidth="8" strokeLinecap="round" />
      <path d="M410 361h140" stroke="#446080" strokeWidth="8" strokeLinecap="round" opacity="0.55" />
      <circle cx="137" cy="251" r="12" fill="#f2ae32" />
      <circle cx="152" cy="235" r="8" fill="#2eb872" />
      <circle cx="602" cy="267" r="12" fill="#e15c9b" />
      <path d="M136 252c-23 13-29 31-22 49" fill="none" stroke="#659b78" strokeWidth="7" strokeLinecap="round" />
    </>
  );
}

function ObservatoryArt({ accent }: { accent: string }) {
  return (
    <>
      <path d="M176 339h197l-31-126H208z" fill="#dbe7f3" stroke="#93a9c1" strokeWidth="5" />
      <path d="M206 217h138l-69-62Z" fill="#6b7d9a" />
      <path d="M237 240h77v98h-77z" fill="#f8fbff" />
      <path d="M250 255h50v65h-50z" fill="#2f6bff" opacity="0.2" />
      <path d="M263 304V270l19 15 18-27" fill="none" stroke={accent} strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M423 298l94-78" stroke="#4c637f" strokeWidth="17" strokeLinecap="round" />
      <path d="M488 221c15-34 51-48 87-30l-19 59c-31 11-55 3-68-29Z" fill="#f2ae32" stroke="#8f7450" strokeWidth="5" />
      <circle cx="430" cy="301" r="24" fill="#2eb872" />
      <circle cx="430" cy="301" r="9" fill="#ffffff" />
      <path d="M418 342h25v39h-25z" fill="#718ca8" />
      <path d="M392 383h77" stroke="#435b77" strokeWidth="9" strokeLinecap="round" />
      <circle cx="561" cy="314" r="17" fill="#2f6bff" opacity="0.75" />
      <circle cx="590" cy="338" r="12" fill="#6b63ff" opacity="0.7" />
    </>
  );
}

function ForecastArt({ accent }: { accent: string }) {
  return (
    <>
      <path d="M147 343h427" stroke="#54708d" strokeWidth="8" strokeLinecap="round" />
      <path d="M175 343V219h370v124" fill="#edf3fa" stroke="#839ab4" strokeWidth="6" />
      <path d="M220 250h79v93h-79zm108 0h79v93h-79zm108 0h79v93h-79z" fill="#ffffff" stroke="#b8c8da" strokeWidth="4" />
      <path d="M231 326l18-28 19 12 21-37" fill="none" stroke="#2f6bff" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M339 326l18-18 19 7 20-32" fill="none" stroke="#2eb872" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M448 326l17-11 20-27 17 13" fill="none" stroke="#d46b4c" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="259" cy="210" r="22" fill={accent} />
      <circle cx="367" cy="210" r="22" fill="#2eb872" />
      <circle cx="475" cy="210" r="22" fill="#d46b4c" />
      <path d="M252 210h14M360 210h14M468 210h14" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" />
      <path d="M247 176v-27m109 27v-27m109 27v-27" stroke="#6c82a0" strokeWidth="6" strokeLinecap="round" />
      <circle cx="247" cy="140" r="8" fill="#6c82a0" />
      <circle cx="356" cy="140" r="8" fill="#6c82a0" />
      <circle cx="465" cy="140" r="8" fill="#6c82a0" />
    </>
  );
}

function NewsroomArt({ accent }: { accent: string }) {
  return (
    <>
      <path d="M164 321c33-87 115-125 193-113 78 12 117 78 104 148H164Z" fill="#cce3ce" />
      <path d="M210 301h91v70h-91z" fill="#fffaf0" stroke="#b8a978" strokeWidth="5" transform="rotate(-7 255 337)" />
      <path d="M315 291h95v78h-95z" fill="#fffaf0" stroke="#b8a978" strokeWidth="5" transform="rotate(7 362 330)" />
      <path d="M225 319h59m-54 17h42m64-16h64m-59 18h42" stroke="#6d7f9a" strokeWidth="7" strokeLinecap="round" />
      <path d="M505 246c-32-36-75-46-115-30l19 67c34 4 70-5 96-37Z" fill="#f5dc89" stroke="#ad945a" strokeWidth="5" />
      <path d="M509 247c-34-1-65 11-94 37" fill="none" stroke={accent} strokeWidth="9" strokeLinecap="round" />
      <path d="M498 286c-20 44-22 75-10 99" fill="none" stroke="#739876" strokeWidth="7" strokeLinecap="round" />
      <circle cx="149" cy="262" r="16" fill="#2eb872" />
      <circle cx="123" cy="278" r="12" fill="#f2ae32" />
      <circle cx="150" cy="304" r="15" fill="#d70015" opacity="0.72" />
      <path d="M151 262c-28-27-35-53-20-80" fill="none" stroke="#6e9e7a" strokeWidth="7" strokeLinecap="round" />
    </>
  );
}

function CompareArt({ accent }: { accent: string }) {
  return (
    <>
      <path d="M156 321h414" stroke="#6c819d" strokeWidth="11" strokeLinecap="round" />
      <path d="M187 320 264 213l77 107m81 0 77-107 77 107" fill="none" stroke="#9aadc3" strokeWidth="9" />
      <path d="M205 263h117m-81-31h55m126 31h117m-81-31h55" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" opacity="0.88" />
      <rect x="222" y="155" width="84" height="52" rx="12" fill="#f8fbff" stroke="#8fa4bd" strokeWidth="5" />
      <rect x="426" y="155" width="84" height="52" rx="12" fill="#f8fbff" stroke="#8fa4bd" strokeWidth="5" />
      <path d="M244 189l14-17 15 9 16-21m171 29 14-22 15 13 16-17" fill="none" stroke={accent} strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="340" cy="259" r="17" fill="#2f6bff" />
      <circle cx="382" cy="259" r="17" fill="#2eb872" />
      <path d="M351 259h20" stroke="#ffffff" strokeWidth="6" strokeLinecap="round" />
      <circle cx="135" cy="282" r="14" fill="#f2ae32" />
      <circle cx="588" cy="281" r="14" fill="#e15c9b" />
    </>
  );
}

function LearnArt({ accent }: { accent: string }) {
  return (
    <>
      <path d="M174 335h387V219H174z" fill="#fbfbf8" stroke="#9aaec6" strokeWidth="5" />
      <path d="M157 220h421l-44-74H201z" fill="#6b83ab" />
      <path d="M224 234h68v101h-68zm106 0h68v101h-68zm106 0h68v101h-68z" fill="#e7eef8" />
      <path d="M239 270h38m-38 20h38m68-20h38m-38 20h38m68-20h38m-38 20h38" stroke="#6d82a0" strokeWidth="7" strokeLinecap="round" />
      <path d="M303 335c8-63 29-101 68-101s60 38 68 101" fill="#f2ae32" opacity="0.88" />
      <path d="M310 335c11-45 30-72 61-72s50 27 61 72" fill="#fff7df" />
      <path d="M339 302c18-21 35-21 53 0" fill="none" stroke={accent} strokeWidth="8" strokeLinecap="round" />
      <circle cx="508" cy="175" r="19" fill="#2eb872" />
      <path d="M497 175h22m-11-11v22" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" />
    </>
  );
}

export function MiniatureArt({ kind, accent }: MiniatureArtProps) {
  const style: ArtStyle = { "--scene-accent": accent };
  return (
    <div className="miniature-art" style={style} aria-hidden="true">
      <svg viewBox="0 0 840 430" role="presentation">
        <defs>
          <linearGradient id={`art-sky-${kind}`} x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#f7fbff" />
            <stop offset="1" stopColor="#dce9f5" />
          </linearGradient>
          <linearGradient id={`art-island-${kind}`} x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#eef6ee" />
            <stop offset="1" stopColor="#b9cfbd" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="840" height="430" rx="38" fill={`url(#art-sky-${kind})`} />
        <circle cx="740" cy="78" r="52" fill="#ffffff" opacity="0.8" />
        <circle cx="740" cy="78" r="38" fill="#dcecff" opacity="0.92" />
        <g transform="translate(0 0)">
          <Island accent={accent} />
          {kind === "exchange" && <ExchangeArt accent={accent} />}
          {kind === "observatory" && <ObservatoryArt accent={accent} />}
          {kind === "forecast" && <ForecastArt accent={accent} />}
          {kind === "newsroom" && <NewsroomArt accent={accent} />}
          {kind === "compare" && <CompareArt accent={accent} />}
          {kind === "learn" && <LearnArt accent={accent} />}
        </g>
      </svg>
    </div>
  );
}
