import axios from 'axios';

const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:3002';

export const flaskService = {
  // Flask 서버에 저장소 인덱싱 요청
  async requestRepositoryIndexing(repoUrl, repositoryInfo) {
    try {
      const response = await axios.post(
        `${FLASK_API_URL}/api/repository/index`,
        {
          repo_url: repoUrl,
          repository_info: repositoryInfo, // 추가 정보 전달
          callback_url: `${
            process.env.EXPRESS_API_URL || 'http://localhost:3001'
          }/api/internal/analysis-complete`, // 콜백 URL 추가
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 30000, // 30초 타임아웃
        }
      );

      return {
        success: true,
        data: response.data,
        status: response.status,
      };
    } catch (error) {
      console.error('Flask 인덱싱 요청 오류:', error.message);

      if (error.response) {
        // Flask 서버에서 응답을 받았지만 오류 상태
        return {
          success: false,
          error: error.response.data?.message || 'Flask 서버 오류',
          status: error.response.status,
          data: error.response.data,
        };
      } else if (error.request) {
        // 요청은 보냈지만 응답을 받지 못함
        return {
          success: false,
          error: 'Flask 서버에 연결할 수 없습니다.',
          status: 503,
        };
      } else {
        // 요청 설정 중 오류
        return {
          success: false,
          error: `요청 설정 오류: ${error.message}`,
          status: 500,
        };
      }
    }
  },

  // Flask 서버에서 저장소 분석 상태 조회
  async getRepositoryAnalysisStatus(repoName) {
    try {
      const response = await axios.get(
        `${FLASK_API_URL}/api/repository/status/${encodeURIComponent(
          repoName
        )}`,
        {
          timeout: 10000, // 10초 타임아웃
        }
      );

      return {
        success: true,
        data: response.data,
        status: response.status,
      };
    } catch (error) {
      console.error('Flask 상태 조회 오류:', error.message);

      if (error.response) {
        return {
          success: false,
          error: error.response.data?.message || 'Flask 서버 오류',
          status: error.response.status,
          data: error.response.data,
        };
      } else if (error.request) {
        return {
          success: false,
          error: 'Flask 서버에 연결할 수 없습니다.',
          status: 503,
        };
      } else {
        return {
          success: false,
          error: `요청 설정 오류: ${error.message}`,
          status: 500,
        };
      }
    }
  },

  // Flask 서버에서 저장소 검색
  async searchRepository(repoName, query, searchType = 'code') {
    try {
      const response = await axios.post(
        `${FLASK_API_URL}/api/repository/search`,
        {
          repo_name: repoName,
          query: query,
          search_type: searchType,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 60000, // 60초 타임아웃 (검색은 시간이 걸릴 수 있음)
        }
      );

      return {
        success: true,
        data: response.data,
        status: response.status,
      };
    } catch (error) {
      console.error('Flask 검색 요청 오류:', error.message);

      if (error.response) {
        return {
          success: false,
          error: error.response.data?.message || 'Flask 서버 오류',
          status: error.response.status,
          data: error.response.data,
        };
      } else if (error.request) {
        return {
          success: false,
          error: 'Flask 서버에 연결할 수 없습니다.',
          status: 503,
        };
      } else {
        return {
          success: false,
          error: `요청 설정 오류: ${error.message}`,
          status: 500,
        };
      }
    }
  },

  // Flask 서버 상태 확인
  async checkFlaskServerHealth() {
    try {
      const response = await axios.get(`${FLASK_API_URL}/`, {
        timeout: 5000, // 5초 타임아웃
      });

      return {
        success: true,
        data: response.data,
        status: response.status,
      };
    } catch (error) {
      return {
        success: false,
        error: 'Flask 서버에 연결할 수 없습니다.',
        status: error.response?.status || 503,
      };
    }
  },
};
