module.exports = {
  apps: [
    {
      name: 'aissue-flask-api',
      script: 'bash',
      args: '-c "/home/ubuntu/miniconda3/envs/py312_aissue/bin/python -m gunicorn run:app --bind 127.0.0.1:3002 --workers 2"',
      interpreter: 'none',
      cwd: '/home/ubuntu/AIssue/AIssue-BE-Flask/',
    },
  ],
};
