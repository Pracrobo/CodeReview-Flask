module.exports = {
  apps: [
    {
      name: "codereview-flask-api",
      script: "bash",
      args: '-c "/home/ubuntu/miniconda3/envs/py312_codereview/bin/python -m gunicorn run:app --bind 127.0.0.1:3002 --timeout 300"',
      interpreter: "none",
      cwd: "/home/ubuntu/CodeReview/CodeReview-Flask/",
    },
  ],
};
