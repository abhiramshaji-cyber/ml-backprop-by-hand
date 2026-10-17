import Lab from "./lab/Lab";

const REPO = "https://github.com/abhiramshaji-cyber/ml-backprop-by-hand";

export default function App() {
  return (
    <main className="page">
      <header className="hero">
        <h1>
          <span className="title">Backprop by hand</span>
          <br />
          <span className="subtitle">
            an autograd engine, an MLP and an optimizer, no libraries
          </span>
        </h1>
        <nav className="links">
          <a href={`${REPO}#how-it-works`}>How it works →</a>
          <a href={REPO}>Source →</a>
        </nav>
      </header>
      <Lab />
    </main>
  );
}
