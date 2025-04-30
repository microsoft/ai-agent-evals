import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "github-markdown-css/github-markdown.css";

export interface IGithubFlavoredMarkdownProps {
  markdownContent: string;
}

const LinkRenderer = (props: any) => {
  return (
    <a {...props} target="_blank" rel="noopener noreferrer">
      {props.children}
    </a>
  );
};

export const GithubFlavoredMarkdown = (props: IGithubFlavoredMarkdownProps) => {
  return (
    <div className="markdown-body" style={{ padding: "2rem" }}>
      <ReactMarkdown
        components={{
          a: LinkRenderer,
        }}
        remarkPlugins={[remarkGfm]}
      >
        {props.markdownContent}
      </ReactMarkdown>
    </div>
  );
};
