import { defineConfig } from "vitepress";

export default defineConfig({
  base: "/EvoSynth/",
  title: "EvoSynth",
  description:
    "Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning",

  head: [
    ["link", { rel: "icon", type: "image/png", href: "/logo.png" }],
  ],

  themeConfig: {
    siteTitle: "EvoSynth",
    logo: "/logo.png",

    nav: [
      { text: "Guide", link: "/guide/introduction" },
      { text: "Algorithms", link: "/algorithms/genetic-algorithm" },
      { text: "API Reference", link: "/reference/controller" },
      { text: "Comparative Analysis", link: "/comparative_analysis" },
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Getting Started",
          collapsed: false,
          items: [
            { text: "Introduction", link: "/guide/introduction" },
            { text: "System Architecture", link: "/guide/architecture" },
            { text: "Installation", link: "/guide/installation" },
          ],
        },
        {
          text: "Core Concepts",
          collapsed: false,
          items: [
            { text: "Meta-Optimizer (UCB1)", link: "/concepts/meta-optimizer" },
            { text: "Universal Latent Space", link: "/concepts/latent-space" },
            {
              text: "Cooperative Co-evolution",
              link: "/concepts/cooperative-coevolution",
            },
            { text: "Composite Scoring", link: "/concepts/composite-scoring" },
          ],
        },
      ],
      "/concepts/": [
        {
          text: "Getting Started",
          collapsed: true,
          items: [
            { text: "Introduction", link: "/guide/introduction" },
            { text: "System Architecture", link: "/guide/architecture" },
            { text: "Installation", link: "/guide/installation" },
          ],
        },
        {
          text: "Core Concepts",
          collapsed: false,
          items: [
            { text: "Meta-Optimizer (UCB1)", link: "/concepts/meta-optimizer" },
            { text: "Universal Latent Space", link: "/concepts/latent-space" },
            {
              text: "Cooperative Co-evolution",
              link: "/concepts/cooperative-coevolution",
            },
            { text: "Composite Scoring", link: "/concepts/composite-scoring" },
          ],
        },
      ],
      "/algorithms/": [
        {
          text: "Optimization Algorithms",
          collapsed: false,
          items: [
            {
              text: "Genetic Algorithm",
              link: "/algorithms/genetic-algorithm",
            },
            {
              text: "Particle Swarm Optimization",
              link: "/algorithms/particle-swarm",
            },
            {
              text: "Differential Evolution",
              link: "/algorithms/differential-evolution",
            },
            { text: "CMA-ES", link: "/algorithms/cmaes" },
          ],
        },
      ],
      "/dashboard/": [
        {
          text: "Live Dashboard",
          collapsed: false,
          items: [
            { text: "Overview", link: "/dashboard/overview" },
            { text: "SSE Protocol", link: "/dashboard/sse-protocol" },
            { text: "Interpreting Data", link: "/dashboard/interpreting-data" },
          ],
        },
      ],
      "/reference/": [
        {
          text: "API Reference",
          collapsed: false,
          items: [
            { text: "DynamicOptimizer", link: "/reference/controller" },
            { text: "Evaluators", link: "/reference/evaluator" },
            { text: "Search Space", link: "/reference/search-space" },
          ],
        },
      ],
      "/developer/": [
        {
          text: "Developer Guide",
          collapsed: false,
          items: [
            {
              text: "Adding a New Optimizer",
              link: "/developer/adding-optimizer",
            },
            { text: "Custom Evaluator", link: "/developer/custom-evaluator" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/paarthureddy/EvoSynth" },
    ],

    footer: {
      message: "Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning",
      copyright: "EvoSynth Research / Capstone Project",
    },

    search: {
      provider: "local",
    },

    outline: {
      level: [2, 3],
      label: "On this page",
    },
  },
});
